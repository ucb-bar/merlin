"""Isolated native-Linux affinity and open-file exec worker; no compiler imports."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import sys


def main():
    image, report = int(sys.argv[1]), int(sys.argv[2])
    expected, cpus, executable = sys.argv[3], json.loads(sys.argv[4]), sys.argv[5]
    os.set_inheritable(report, False)
    os.set_inheritable(image, False)

    def observe(record):
        payload = json.dumps(record, sort_keys=True).encode() + b"\n"
        if len(payload) > 16384 or os.write(report, payload) != len(payload):
            raise RuntimeError("incomplete Linux execution observation")

    try:
        info = os.fstat(image)
        if not stat.S_ISREG(info.st_mode):
            raise ValueError("selected Linux image is not a regular file")
        with os.fdopen(os.dup(image), "rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != expected:
            raise ValueError("selected Linux image changed before exec")
        os.lseek(image, 0, os.SEEK_SET)
        if (
            type(cpus) is not list
            or not cpus
            or len(cpus) > 1024
            or any(type(cpu) is not int or cpu < 0 for cpu in cpus)
            or len(set(cpus)) != len(cpus)
        ):
            raise ValueError("Linux affinity needs a bounded distinct CPU roster")
        if not set(cpus) <= os.sched_getaffinity(0):
            raise ValueError("selected Linux CPUs are no longer available")
        os.sched_setaffinity(0, cpus)
        affinity = sorted(os.sched_getaffinity(0))
        if affinity != cpus:
            raise ValueError("kernel did not apply the selected Linux affinity")
        observe(
            {
                "stage": "exec_requested",
                "pid": os.getpid(),
                "affinity": affinity,
                "image_sha256": actual,
                "device": info.st_dev,
                "inode": info.st_ino,
                "bytes": info.st_size,
            }
        )
        os.execve(image, [executable, *sys.argv[6:]], os.environ)
    except Exception as error:
        observe({"stage": "exec_failed", "error": f"{type(error).__name__}: {error}"})
        print(f"Linux execution refused: {error}", file=sys.stderr)
        return 126


if __name__ == "__main__":
    raise SystemExit(main())
