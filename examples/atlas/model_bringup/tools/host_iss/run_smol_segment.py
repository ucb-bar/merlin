"""Execute one admitted full Smol segment in a fresh RV64 guest."""
import argparse
from pathlib import Path
import struct
import sys

sys.path.insert(0, str(Path(__file__).absolute().parents[1]))
import kernel_library  # noqa: E402,F401
from atlas_path_guard import checked
import atlas_host_layout as hal
from segment_runner import SegmentRunner
from smol_full_phases import execution_profile
import smol_segment_contract as contract

WORKLOAD = "smol-full"
REQUEST_SYMBOL = "smol_full_segment_request"
RESULT_NAME = "smol-segment"
CHECK_FAILURE_SOURCES = True
PIN_MODULES = ("smol_full_plan", "smol_full_phases", "atlas_program", "placement", "atlas_tile_arithmetic")
PIN_PATHS = ("chipyard/arena.py",)
VERIFY_PROGRAM_PINS = True
CORRUPT_REGIONS = None
STOP_ERROR = "guest trapped or rejected boundary during segment"


def receipt_fields():
    return dict(workload="smol-full", full_soc_executed=False,
                scope="one semantic phase through normal RV64 entry and a fresh Model Device")


def initialize(remote, image):
    stack_pointer = remote.reg(2)
    assert stack_pointer % 16 == 0 and (image.symbols["__stack_bottom"]["address"]
        <= stack_pointer < image.symbols["__stack_top"]["address"])
    initialized_host = remote.read(image.symbols["smol_full_host"]["address"], 112)
    assert struct.unpack_from("<3Q", initialized_host) == (
        hal.ATLAS_IMEM_BASE, hal.ATLAS_CSR_BASE, hal.ATLAS_IMEM_BYTES)
    assert struct.unpack_from("<3Q", initialized_host, 24) == (
        image.symbols["atlas_host_direct_read32"]["address"],
        image.symbols["atlas_host_direct_write32"]["address"], 0)
    assert initialized_host[48:72] == struct.pack("<I4sQQ", 1, bytes(4), 0, 0)
    assert initialized_host[72:] == bytes(40), "segment entry host is not freshly initialized"
    assert remote.read(image.symbols["atlas_workload_mailbox"]["address"], 128) == struct.pack(
        "<16Q", contract.full.MAGIC, *([0]*15))
    # Static segment admission already binds this extent to the execution profile.
    observations = image.symbols["smol_full_segment_observations"]
    assert remote.read(observations["address"], observations["size"]) == bytes(observations["size"])
    assert remote.read(image.symbols["tohost"]["address"], 8) == bytes(8)
    assert remote.read(image.symbols["fromhost"]["address"], 8) == bytes(8)
    return initialized_host


def capture_stop(remote, image, out, pc, receipt):
    receipt["stop_pc"] = pc
    receipt["terminal_mailbox_words"] = struct.unpack("<16Q", remote.read(
        image.symbols["atlas_workload_mailbox"]["address"], 128))
    receipt["terminal_tohost"] = struct.unpack("<Q", remote.read(image.symbols["tohost"]["address"], 8))[0]
    receipt["terminal_uart"] = (out / "uart.log").read_bytes().decode("ascii", errors="replace")
    lines = [line for line in receipt["terminal_uart"].splitlines() if line.startswith("ATLAS_KERNEL_")]
    receipt["guest_fail_before_pass"] = any(line.startswith("ATLAS_KERNEL_FAIL ") for line in lines) and not any(
        line.startswith("ATLAS_KERNEL_PASS ") for line in lines)


def verify_state_after(remote, image, chain, data, index, completed, initial, receipt, launch_range):
    receipt.update(verify_state(remote.read, image, chain, data, index, completed, initial))


def verify_observations(raw, completed, launch_range, kernel_count=2915):
    """Bind the preserved guest cycle deltas to only genuinely completed kernels."""
    assert len(raw) == 4 * kernel_count
    assert [row["index"] for row in completed] == list(range(*launch_range))
    expected = [0] * kernel_count
    for row in completed:
        assert type(row["cycles"]) is int and 0 < row["cycles"] <= contract.full.CYCLE_CAP
        expected[row["index"]] = row["cycles"]
    assert struct.unpack("<" + str(kernel_count) + "I", raw) == tuple(expected), "guest cycle observations differ"
    return dict(observation_bytes=len(raw), observation_sha256=contract.full.digest(raw),
                actual_observation_cycles=sum(expected), observation_launch_range=list(launch_range))


def verify_state(read, image, chain, data, index, completed, initialized_host):
    """Check live ownership separately from the complete semantic byte comparison."""
    profile = execution_profile(data)
    def actual(name, size):
        return read(image.symbols[name]["address"], size)
    launch_start, launch_end = (b["launch_cursor"] for b in chain.boundaries[index:index + 2])
    cursor = struct.unpack("<10I", chain.read(chain.boundaries[index + 1]["regions"]["smol_full_phase_state"]))
    attention_cursor = index + 1 if profile.attention_count else 0
    assert cursor[0:2] == (1, index + 1) and cursor[5]+attention_cursor == cursor[7] == launch_end
    observation = verify_observations(actual("smol_full_segment_observations", profile.cycle_bytes), completed,
                                      (launch_start, launch_end), profile.launch_count)
    assert actual("smol_full_cycles", profile.cycle_bytes) == bytes(profile.cycle_bytes), "semantic cycle array was not canonicalized"
    raw = actual("smol_full_raw", 184)
    current = struct.unpack("<15Q24sIIi4s3Q", raw)
    initial = struct.unpack("<15Q24sIIi4s3Q", image.bytes_at(image.symbols["smol_full_raw"]["address"], 184))
    assert current[5:7] == (cursor[5], cursor[6])
    assert all(current[i] == initial[i] for i in (*range(5), *range(7,15))), "raw immutable ownership changed"
    last_event = profile.events()[launch_end-1]
    last = data["attention_kernels" if last_event["kind"] == "attention" else "kernels"][last_event["kernel"]]
    assert current[15] == struct.pack("<QQI4s", chain.placement["arena_base"], last["total_dram_bytes"], 2, bytes(4))
    image_index = cursor[3] if cursor[2] == 0 else 1
    assert current[16:20] == (image_index, 0, 0, bytes(4)) and current[20:22] == initial[20:22] and current[22] == attention_cursor, "raw image/error state differs"
    host = actual("smol_full_host", 112)
    assert host[:72] == initialized_host[:72], "host map/MMIO/cache ownership changed"
    assert struct.unpack_from("<QQ", host, 72) == (0, 0)
    flags = struct.unpack_from("<6I", host, 88)
    assert flags[:3] == (1, 0, 0) and flags[4:] == (1, launch_end), "host load/drain/global sequence differs"
    assert actual("smol_full_segment_request", 8) == struct.pack("<Q", index)
    assert actual("tohost", 8) == struct.pack("<Q", 1) and actual("fromhost", 8) == bytes(8)
    mailbox = struct.unpack("<16Q", actual("atlas_workload_mailbox", 128))
    # Driver ABI completion fields are pinned independently of the semantic captures.
    previous = struct.unpack("<10I", chain.read(chain.boundaries[index]["regions"]["smol_full_phase_state"]))
    graph = struct.unpack("<4I", chain.read(chain.boundaries[index + 1]["regions"]["smol_full_graph_state"]))
    expected_mailbox = (contract.full.MAGIC, 2, launch_end-launch_start, cursor[6]-previous[6],
                        graph[0], graph[1], 1, 0, 0, cursor[5], cursor[6], launch_end, 0, 0, 0, 0)
    assert mailbox == expected_mailbox, "actual segment mailbox differs"
    return dict(**observation, mailbox_words=mailbox, raw_context_sha256=contract.full.digest(raw),
                host_global_sequence=launch_end, logical_call_cursor=cursor[6])


def execute(args):
    return SegmentRunner(args, sys.modules[__name__]).execute()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bundle-root", "model-root", "qemu", "out"):
        parser.add_argument("--" + name, type=checked, required=True)
    parser.add_argument("--segment-index", type=int, required=True, choices=range(200), metavar="0..199")
    parser.add_argument("--wall-seconds", type=int, required=True)
    parser.add_argument("--sifive-uart-bridge", action="store_true")
    parser.add_argument("--corrupt-region", help="explicit semantic-region corruption control after restore")
    parser.add_argument("--corrupt-offset", type=int, default=1)
    args = parser.parse_args()
    if not 0 < args.wall_seconds <= 28800:
        parser.error("explicit segment wall cap must be1..28800 seconds")
    raise SystemExit(execute(args))


if __name__ == "__main__":
    if not __debug__:
        raise SystemExit("Python -O is unsupported")
    main()
