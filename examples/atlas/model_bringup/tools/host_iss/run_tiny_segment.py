"""Execute one admitted full Tiny segment in a fresh RV64 guest."""
import argparse
from pathlib import Path
import struct
import sys

sys.path.insert(0, str(Path(__file__).absolute().parents[1]))
import kernel_library  # noqa: E402,F401
from atlas_path_guard import checked
from segment_runner import SegmentRunner
import tiny_segment_contract as contract

WORKLOAD = "tiny-full"
REQUEST_SYMBOL = "tiny_full_segment_request"
RESULT_NAME = "tiny-segment"
CHECK_FAILURE_SOURCES = False
PIN_MODULES = ("chipyard.arena",)
PIN_PATHS = ()
VERIFY_PROGRAM_PINS = False
CORRUPT_REGIONS = ("hidden_0", "hidden_1")
STOP_ERROR = "guest trapped during segment"


def receipt_fields():
    return dict(captured_replies=False)


def initialize(remote, image):
    stack_pointer = remote.reg(2)
    assert stack_pointer % 16 == 0 and (image.symbols["__stack_bottom"]["address"]
        <= stack_pointer < image.symbols["__stack_top"]["address"])


def capture_stop(remote, image, out, pc, receipt):
    pass


def verify_state_after(remote, image, chain, data, index, completed, initial, receipt, launch_range):
    end_control = struct.unpack("<8Q", chain.read(chain.boundaries[index + 1]["regions"]["control"]))
    mailbox = struct.unpack("<16Q", remote.read(image.symbols["tiny_full_mailbox"]["address"], 128))
    assert mailbox == (contract.full.MAGIC, 2, launch_range[1], launch_range[1] - 1, 0, 0,
        int(index // 23 > 0), index + 1, end_control[2], end_control[4],
        0, 0, 0, 0, 0, 0), "segment actual mailbox completion"
    assert remote.read(image.symbols["tohost"]["address"], 8) == struct.pack("<Q", 1)
    assert remote.read(image.symbols["fromhost"]["address"], 8) == bytes(8)
    # Mutable cursors/state are checked in addition to every serialized semantic byte.
    state = struct.unpack("<5QII", remote.read(image.symbols["tiny_full_state_global"]["address"], 48))
    initial = struct.unpack("<5QII", image.bytes_at(image.symbols["tiny_full_state_global"]["address"], 48))
    end_control = struct.unpack("<8Q", chain.read(chain.boundaries[index + 1]["regions"]["control"]))
    assert state[:4] == initial[:4] and state[4:] == end_control[2:5]
    context = struct.unpack("<9Q", remote.read(image.symbols["tiny_full_projection_state"]["address"], 72))
    original = struct.unpack("<9Q", image.bytes_at(image.symbols["tiny_full_projection_state"]["address"], 72))
    assert context[2] == launch_range[1] and context[7] == end_control[6]
    assert all(context[i] == original[i] for i in (0, 1, 3, 4, 5, 6, 8))
    host = remote.read(image.symbols["tiny_full_host"]["address"], 112)
    assert struct.unpack_from("<QQ", host, 72) == (0, 0)
    assert struct.unpack_from("<6I", host, 88)[1:3] == (0, 0)
    assert struct.unpack_from("<I", host, 108)[0] == launch_range[1]


def execute(args):
    return SegmentRunner(args, sys.modules[__name__]).execute()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bundle-root", "model-root", "qemu", "out"):
        parser.add_argument("--" + name, type=checked, required=True)
    parser.add_argument("--segment-index", type=int, required=True, choices=range(69))
    parser.add_argument("--wall-seconds", type=int, required=True)
    parser.add_argument("--sifive-uart-bridge", action="store_true")
    parser.add_argument("--corrupt-region", choices=("hidden_0", "hidden_1"))
    parser.add_argument("--corrupt-offset", type=int, default=1)
    args = parser.parse_args()
    if not 0 < args.wall_seconds <= 43200:
        parser.error("explicit segment wall cap must be1..43200seconds")
    raise SystemExit(execute(args))


if __name__ == "__main__":
    if not __debug__:
        raise SystemExit("Python -O is unsupported")
    main()
