"""Segment admission rejects incomplete live state and incorrect device evidence."""
import copy
from pathlib import Path
import struct
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest import mock

TOOLS = Path(__file__).absolute().parents[1]/"tools"
sys.path[:0] = [str(TOOLS), str(TOOLS/"host_iss")]
import smol_segment_contract as contract
import run_smol_segment as workload
from segment_runner import SegmentRunner
from smol_full_phases import V1, V2, segments
from smol_full_plan import panels
from full_bundle import KernelAssets


class GuestFixture:
    """Bound every guest read to its admitted symbol, including both cycle arrays."""
    def __init__(self, profile, index):
        self.profile, self.index = profile, index
        self.phase = phase = segments(profile)[index]
        self.data = dict(kernel_count=2715, kernels=panels())
        for row in self.data["kernels"]:
            row.update(total_dram_bytes=1024+row["index"], original="projection-"+str(row["index"]),
                       expected="expected", requested_dma={})
        if profile.attention_count:
            self.data.update(attention_count=200, device_event_count=profile.launch_count,
                device_events=profile.events(), attention_kernels=[dict(index=e["kernel"], phase=e["kernel"],
                    before_call=e["before_call"], total_dram_bytes=8192+e["kernel"],
                    original="attention-"+str(e["kernel"]), expected="expected", requested_dma={})
                    for e in profile.events() if e["kind"] == "attention"])
        self.symbols, self.memory, self.reads = {}, {}, []
        for name, size in (("smol_full_host", 112), ("atlas_workload_mailbox", 128),
                           ("smol_full_segment_observations", profile.cycle_bytes), ("tohost", 8),
                           ("fromhost", 8), ("smol_full_cycles", profile.cycle_bytes),
                           ("smol_full_raw", 184), ("smol_full_segment_request", 8)):
            address = 0x81000000 + len(self.symbols)*65536
            self.symbols[name] = dict(address=address, size=size)
            self.memory[address] = bytes(size)
        for name, address in (("__stack_bottom", 0x83000000), ("__stack_top", 0x83010000),
                              ("atlas_host_direct_read32", 100), ("atlas_host_direct_write32", 200)):
            self.symbols[name] = dict(address=address)
        self.host = struct.pack("<6QI4sQQ40s", workload.hal.ATLAS_IMEM_BASE, workload.hal.ATLAS_CSR_BASE,
            workload.hal.ATLAS_IMEM_BYTES, 100, 200, 0, 1, bytes(4), 0, 0, bytes(40))
        self.store("smol_full_host", self.host)
        self.store("atlas_workload_mailbox", struct.pack("<16Q", contract.full.MAGIC, *([0]*15)))
        self.initial_context = [*range(15), bytes(24), 0, 0, 0, bytes(4),
                                6 if profile.attention_count else 0, profile.attention_count, 0]
        self.initial_raw = struct.pack("<15Q24sIIi4s3Q", *self.initial_context)
        self.image = SimpleNamespace(symbols=self.symbols, entry=500,
            bytes_at=lambda address, size: self.initial_raw if address == self.symbols["smol_full_raw"]["address"]
                     and size == 184 else self.read(address, size))
        self.remote = SimpleNamespace(reg=lambda index: 0x8300fff0, read=self.read,
            request=lambda _: b"S05", set_reg=mock.Mock(), breakpoint=mock.Mock(), resume=lambda: (600, b"S05"))
        self.launch_range = phase["launch_start"], phase["launch_end"]
        self.completed = [dict(index=i, cycles=i+1) for i in range(*self.launch_range)]
        cursor = (1, index+1, {"vision": 0, "prefix": 1, "expert": 2}[phase["kind"]], phase["ordinal"],
                  phase["layer"], phase["panel_end"], phase["call_end"], phase["launch_end"], 0, 0)
        previous = (1, index, 0, 0, 0, phase["panel_start"], phase["call_start"], phase["launch_start"], 0, 0)
        self.captures = dict(previous=struct.pack("<10I", *previous), cursor=struct.pack("<10I", *cursor),
                             graph=struct.pack("<4I", 3, 4, 0, 0))
        self.chain = SimpleNamespace(placement=dict(arena_base=0x90000000), read=self.captures.__getitem__,
            boundaries=[{} for _ in range(index)]+[
                dict(launch_cursor=phase["launch_start"], regions=dict(smol_full_phase_state="previous")),
                dict(launch_cursor=phase["launch_end"], regions=dict(smol_full_phase_state="cursor",
                    smol_full_graph_state="graph"))])

    def store(self, name, raw):
        assert len(raw) == self.symbols[name]["size"]
        self.memory[self.symbols[name]["address"]] = raw

    def read(self, address, size):
        raw = self.memory[address]
        assert 0 <= size <= len(raw), "guest read exceeds admitted symbol"
        self.reads.append((address, size))
        return raw[:size]

    def finish(self):
        phase, profile = self.phase, self.profile
        observations = [0]*profile.launch_count
        for row in self.completed:
            observations[row["index"]] = row["cycles"]
        self.store("smol_full_segment_observations", struct.pack("<"+str(profile.launch_count)+"I", *observations))
        cursor = struct.unpack("<10I", self.captures["cursor"])
        context = list(self.initial_context)
        context[5:7] = phase["panel_end"], phase["call_end"]
        event = profile.events()[phase["launch_end"]-1]
        last = self.data["attention_kernels" if event["kind"] == "attention" else "kernels"][event["kernel"]]
        context[15] = struct.pack("<QQI4s", self.chain.placement["arena_base"], last["total_dram_bytes"], 2, bytes(4))
        context[16], context[22] = cursor[3] if cursor[2] == 0 else 1, self.index+1 if profile.attention_count else 0
        self.store("smol_full_raw", struct.pack("<15Q24sIIi4s3Q", *context))
        host = bytearray(self.host)
        struct.pack_into("<6I", host, 88, 1, 0, 0, 0, 1, phase["launch_end"])
        self.store("smol_full_host", bytes(host))
        self.store("smol_full_segment_request", struct.pack("<Q", self.index))
        self.store("tohost", struct.pack("<Q", 1))
        self.store("atlas_workload_mailbox", struct.pack("<16Q", contract.full.MAGIC, 2,
            phase["launch_end"]-phase["launch_start"], phase["call_end"]-phase["call_start"],
            3, 4, 1, 0, 0, phase["panel_end"], phase["call_end"], phase["launch_end"], 0, 0, 0, 0))


class SmolSegmentTest(unittest.TestCase):
    def test_complete_runner_binds_receipts_to_typed_global_events(self):
        for profile in (V1, V2):
            for index in (0, 51, 199):
                with self.subTest(profile=profile.name, index=index), tempfile.TemporaryDirectory() as directory:
                    guest = GuestFixture(profile, index)
                    runner = SegmentRunner.__new__(SegmentRunner)
                    runner.initial = workload.initialize(guest.remote, guest.image)
                    guest.finish()
                    runner.remote, runner.image, runner.chain, runner.data = (
                        guest.remote, guest.image, guest.chain, guest.data)
                    runner.device = SimpleNamespace(completed=guest.completed, launch_count=len(guest.completed))
                    runner.bundle = SimpleNamespace(kernels=guest.data["kernels"], kernel_count=profile.launch_count,
                        device_events=guest.data.get("device_events"), attention_kernels=guest.data.get("attention_kernels", []),
                        read=lambda member: member.encode(), sums={"expected": "e"*64})
                    runner.bundle.kernel_paths = KernelAssets.kernel_paths.__get__(runner.bundle)
                    runner.workload, runner.contract, runner.launch_range = workload, contract, guest.launch_range
                    runner.count, runner.args, runner.receipt = len(guest.completed), SimpleNamespace(segment_index=index), {}
                    runner.pins = {}
                    runner.out, runner.deadline, runner.verify_sources = Path(directory), time.monotonic()+60, mock.Mock()
                    guest.chain.compare = mock.Mock(return_value=[])
                    (runner.out/"uart.log").write_bytes(b"fixture")
                    with mock.patch.object(contract, "verify_kernel") as verify, \
                         mock.patch("segment_runner.verify_uart", return_value={}):
                        runner.verify_completion()
                    self.assertEqual(runner.receipt["status"], "PASS_SEGMENT_WITH_MODEL_DEVICE")
                    self.assertEqual(runner.receipt["observation_bytes"], profile.cycle_bytes)
                    self.assertEqual(verify.call_count, len(guest.completed))
                    for call, event in zip(verify.call_args_list, profile.events()[slice(*guest.launch_range)]):
                        expected = ("attention-" if event["kind"] == "attention" else "projection-")+str(event["kernel"])
                        self.assertEqual(call.args[1]["original"], expected)
                        self.assertEqual(call.args[1]["program"], expected.encode())
                        self.assertEqual(call.args[2], event["index"])

    def test_guest_entry_and_complete_phase_verification_use_admitted_profile(self):
        for profile in (V1, V2):
            for index in (0, 51, 199):
                with self.subTest(profile=profile.name, index=index):
                    guest = GuestFixture(profile, index)
                    runner = SegmentRunner.__new__(SegmentRunner)
                    runner.remote, runner.image = guest.remote, guest.image
                    runner.workload, runner.uart = workload, mock.Mock()
                    runner.ready, runner.stops, runner.sites, runner.receipt = {600: 4}, {700: 4}, {}, {}
                    runner.enter_guest()
                    self.assertEqual(runner.initial, guest.host)
                    self.assertTrue(runner.receipt["host_elf_executed"])
                    guest.finish()
                    workload.verify_state_after(guest.remote, guest.image, guest.chain, guest.data, index,
                        guest.completed, runner.initial, runner.receipt, guest.launch_range)
                    self.assertEqual(runner.receipt["observation_bytes"], profile.cycle_bytes)
                    self.assertEqual(runner.receipt["actual_observation_cycles"], sum(r["cycles"] for r in guest.completed))
                    self.assertEqual(runner.receipt["observation_launch_range"], list(guest.launch_range))
                    self.assertEqual(runner.receipt["host_global_sequence"], guest.launch_range[1])
                    for name in ("smol_full_segment_observations", "smol_full_cycles"):
                        address = guest.symbols[name]["address"]
                        self.assertTrue(any(a == address for a, _ in guest.reads))
                        self.assertTrue(all(size == profile.cycle_bytes for a, size in guest.reads if a == address))

    def test_profile_specific_cursor_and_cycle_corruption_fail(self):
        for profile in (V1, V2):
            for mutation in ("projection", "attention", "observation", "semantic_cycles"):
                with self.subTest(profile=profile.name, mutation=mutation):
                    guest = GuestFixture(profile, 51)
                    initial = workload.initialize(guest.remote, guest.image)
                    guest.finish()
                    if mutation == "projection":
                        cursor = list(struct.unpack("<10I", guest.captures["cursor"]))
                        cursor[5] += 1
                        guest.captures["cursor"] = struct.pack("<10I", *cursor)
                    elif mutation == "attention":
                        raw = bytearray(guest.memory[guest.symbols["smol_full_raw"]["address"]])
                        struct.pack_into("<Q", raw, 176, 1 if profile == V1 else 0)
                        guest.store("smol_full_raw", bytes(raw))
                    else:
                        name = "smol_full_segment_observations" if mutation == "observation" else "smol_full_cycles"
                        raw = bytearray(guest.memory[guest.symbols[name]["address"]])
                        raw[0] = 1
                        guest.store(name, bytes(raw))
                    with self.assertRaises(AssertionError):
                        workload.verify_state_after(guest.remote, guest.image, guest.chain, guest.data, guest.index,
                            guest.completed, initial, {}, guest.launch_range)

    def test_debugger_rendezvous_does_not_halt_default_full_program(self):
        instructions = {
            "smol_full_segment_ready": [(100, 4, 0x13), (104, 2, 0x8082)],
            "smol_full_segment_done": [(200, 4, 0x10500073)],
            "smol_full_segment_failed": [(300, 4, 0x10500073)],
        }
        image = SimpleNamespace(instructions=instructions.__getitem__)
        with mock.patch.object(contract.full, "stop_sites", return_value={400: 4}):
            ready, stops, success = contract.runtime_sites(image)
            self.assertEqual((ready, success), ({100: 4}, {200: 4}))
            self.assertEqual(set(stops), {200, 300, 400})
            instructions["smol_full_segment_ready"].append((106, 4, 0x10500073))
            with self.assertRaisesRegex(ValueError, "must not halt"):
                contract.runtime_sites(image)

    def test_qemu_exit_race_is_reaped_and_timeout_escalates(self):
        import segment_runner as runner
        for race in (True, False):
            process = mock.Mock(pid=123)
            process.poll.side_effect = [None, 0]
            if not race:
                process.wait.side_effect = [runner.subprocess.TimeoutExpired("qemu", 10), 0]
            with mock.patch.object(runner.os, "killpg", side_effect=ProcessLookupError if race else None) as kill:
                self.assertTrue(runner.reap_qemu(process))
                self.assertEqual(kill.call_count, 1 if race else 2)
                self.assertEqual(process.wait.call_count, 1 if race else 2)

    def test_phase_partition_covers_all_calls_in_source_order(self):
        phases = contract.partitions()
        self.assertEqual([sum(p["identity"][0] == kind for p in phases) for kind in range(3)], [24, 16, 160])
        self.assertEqual([i for p in phases for i in range(p["call_start"], p["call_end"])], list(range(1421)))
        self.assertEqual(phases[0]["identity"], (0, 0, 0))
        self.assertEqual(phases[-1]["identity"], (2, 9, 15))

    def evidence(self):
        item = dict(program=struct.pack("<2I", 0x13, 0x73), expected_sha256="e"*64,
                    initial_sha256="a"*64, requested_dma=dict(load_bytes=128, store_bytes=64))
        summary = dict(index=39, sequence=40, reset_count=1, cycles=1234, exact=True, drained=True,
            program_sha256=contract.full.digest(item["program"][:-4]+contract.full.EPILOGUE),
            host_input_sha256="a"*64, actual_output_sha256="e"*64, expected_output_sha256="e"*64,
            observed_dma=dict(load_bytes=128, store_bytes=64, response_beats=6))
        return item, summary

    def test_actual_global_ledger_and_payload_binding(self):
        item, summary = self.evidence()
        contract.verify_kernel(summary, item, 39)
        for key, value in (("index", 0), ("sequence", 1), ("reset_count", 0), ("cycles", 0),
                           ("exact", False), ("drained", False), ("cycles", True),
                           ("host_input_sha256", "b"*64), ("actual_output_sha256", "c"*64),
                           ("program_sha256", "d"*64), ("observed_dma", {})):
            altered = copy.deepcopy(summary)
            altered[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                contract.verify_kernel(altered, item, 39)

    def test_readonly_tables_require_file_backing(self):
        image = SimpleNamespace(segments=[dict(address=1024, memory_bytes=64, file_bytes=32, flags=4)])
        contract.span(image, 1024, 32, False)
        with self.assertRaises(ValueError):
            contract.span(image, 1024, 33, False)
        with self.assertRaises(ValueError):
            contract.span(image, 1024, 32, True)


if __name__ == "__main__":
    unittest.main()
