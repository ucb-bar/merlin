"""Tiny synthetic child only: no network, SDK, compiler or descendants."""
import os
from pathlib import Path
import signal
import sys


def block():
    while True:
        signal.pause()


mode = sys.argv[1]
Path("CONTROL-STARTED.txt").write_text(mode)
if mode == "normal":
    Path("COMPLETE.txt").write_text("complete including cleanup")
elif mode == "open":
    os.mkfifo("synthetic-open.fifo", 0o600)
    os.open("synthetic-open.fifo", os.O_RDONLY)
elif mode == "read":
    Path("partial.bin").write_bytes(b"retained before blocked read")
    reader, writer = os.pipe()
    os.read(reader, 1)
elif mode == "close":
    Path("partial.bin").write_bytes(b"retained before blocking close")
    class Response:
        def close(self):
            Path("CLOSE-ENTERED.txt").write_text("cleanup after first failure")
            block()
    try:
        raise ValueError("synthetic first child error only in memory")
    finally:
        Response().close()
elif mode == "kill":
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    Path("IGNORE-TERM-READY.txt").write_text("only this child")
    block()
elif mode == "nonzero":
    raise SystemExit(7)
else:
    raise ValueError("unknown tiny control")
