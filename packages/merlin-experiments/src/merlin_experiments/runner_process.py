"""Bounded process-group inspection and shutdown for frozen experiment plans."""
from __future__ import annotations

import os
import signal
import subprocess
from pathlib import Path


def _process_active(pid: int | None) -> bool:
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    # A reaped-by-another-supervisor zombie no longer owns any working process.
    stat = Path(f"/proc/{pid}/stat")
    try:
        return stat.read_text().rpartition(") ")[2].split()[0] != "Z"
    except (OSError, IndexError):
        return True


def _process_group_active(group: int) -> bool:
    """A driver exit is not proof that its workers exited. Unknown stays active."""
    processes = Path("/proc")
    if not processes.is_dir():
        return True
    try:
        for path in processes.iterdir():
            if not path.name.isdigit():
                continue
            try:
                fields = (path / "stat").read_text().rpartition(") ")[2].split()
                if int(fields[2]) == group and fields[0] != "Z":
                    return True
            except FileNotFoundError:
                continue  # a process exited during inspection
            except (OSError, IndexError, ValueError):
                return True
    except OSError:
        return True
    return False


def _stop_process_group(process: subprocess.Popen) -> None:
    """Best-effort shutdown, never proof that an abnormal run left no descendants."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=1)
    except subprocess.TimeoutExpired:
        pass
    finally:
        # Workers may outlive their driver; always signal the whole private group.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    try:
        process.wait(timeout=1)
    except subprocess.TimeoutExpired:
        pass
