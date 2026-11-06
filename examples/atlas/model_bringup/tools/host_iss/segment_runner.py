"""QEMU execution, live-state restoration, and receipts for model segments."""
import datetime
import json
import os
from pathlib import Path
import resource
import signal
import struct
import subprocess
import sys
import time

import kernel_library  # noqa: E402,F401
from atlas_path_guard import checked
from atlas_tooling import sha256
from kernel_library import code_sources
import atlas_host_layout as hal
from elf_image import Image, mmio_sites
from full_bundle import FullBundle
from gdb_remote import Remote
from model_device import Device
from process_cleanup import reap_qemu
from result_lines import verify as verify_uart
from segment_uart import Transport
from run_model_tile import configure_model, model_binding


class SegmentRunner:
    def __init__(self, args, workload):
        self.args, self.workload = args, workload
        self.contract = workload.contract
        self.out = checked(args.out)
        self.out.mkdir(parents=True, exist_ok=False)
        self.started = time.monotonic()
        self.deadline = time.monotonic() + args.wall_seconds
        self.receipt = dict(
            schema="atlas.segment_execution.v1", status="FAILED", segmented=True,
            continuous=False, fresh_device=False, host_elf_executed=False,
            source_bookends_equal=False, hardware_execution=False,
            segment_index=args.segment_index, command=[sys.executable, *sys.argv], kernels=[],
            **workload.receipt_fields())
        self.remote = self.process = self.device = self.uart = None
        self.pins, self.previous_cwd = {}, Path.cwd()

    def progress(self, stage):
        state = dict(
            stage=stage, status=self.receipt["status"] if stage == "finished" else "RUNNING",
            runner_pid=os.getpid(), launch_range=self.receipt.get("launch_range"),
            fresh_device=self.receipt.get("fresh_device", False),
            qemu_pid=self.process.pid if self.process else None,
            segment_index=self.args.segment_index,
            updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            elapsed_seconds=time.monotonic() - self.started,
            completed_launches=len(self.device.completed) if self.device else 0,
            completed_device_cycles=sum(k["cycles"] for k in self.device.completed) if self.device else 0,
            runner_max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        temporary = self.out / "run-state.tmp"
        temporary.write_text(json.dumps(state, indent=2) + "\n")
        temporary.replace(self.out / "run-state.json")

    def check_inputs(self):
        args, pins = self.args, self.pins
        root = checked(args.bundle_root)
        self.elf, self.qemu, self.model_root = (
            checked(path) for path in (root / "host/host.elf", args.qemu, args.model_root))
        Transport.pin(pins)
        self.pin_sources()
        self.bundle = FullBundle(root, self.workload.WORKLOAD)
        self.image = Image(self.bundle.read("host/host.elf"))
        self.uart = Transport(self.bundle.build["uart"],
                              getattr(args, "sifive_uart_bridge", False), self.image, self.out)
        before = model_binding(self.model_root)
        _, _, assembler, _ = configure_model(self.model_root)
        self.binding = binding = model_binding(self.model_root)
        assert binding["tracked_source_diff_exit"] == before["tracked_source_diff_exit"] == 0
        assert binding["revision"] == before["revision"]
        assert binding["curve_sha256"] == before["curve_sha256"]
        profile = dict(revision=binding["revision"], backend="curve",
                       curve_sha256=binding["curve_sha256"])
        self.chain, self.data = self.contract.validate_bundle(self.bundle, self.image, assembler, profile)
        assert all(sha256(checked(path)) == value for path, value in pins.items()), "admission helper changed"
        assert model_binding(self.model_root) == binding, "admission model source changed"
        assert 0 <= args.segment_index < len(self.chain.boundaries) - 1
        self.launch_range = tuple(boundary["launch_cursor"] for boundary in
                                  self.chain.boundaries[args.segment_index:args.segment_index + 2])
        self.count = self.launch_range[1] - self.launch_range[0]
        self.ready, self.stops, self.success = self.contract.runtime_sites(self.image)
        self.sites = mmio_sites(self.image)
        self.check_memory()
        self.receipt.update(
            launch_range=list(self.launch_range), elf_sha256=self.bundle.sums["host/host.elf"],
            contract_sha256=self.bundle.sums[self.bundle.provenance["contract"]], model_profile=profile,
            chain_sha256sums_sha256=self.chain.checksums_sha256, full_static_admission=True,
            live_regions=self.chain.data["live_regions"], model_binding=binding)

    def pin_sources(self):
        tools = Path(__file__).absolute().parents[1]
        modules = ("process_cleanup", "segment_boundary", "model_device", "elf_image", "gdb_remote",
                   "full_bundle", "bundle", "result_lines", "run_model_tile", "atlas_host_layout",
                   "atlas_path_guard", "atlas_tooling", *self.workload.PIN_MODULES)
        for module in (self.contract, self.contract.full, *(sys.modules[name] for name in modules)):
            path = checked(Path(module.__file__).absolute())
            self.pins[str(path)] = sha256(path)
        for name, expected in code_sources(traffic_reuse=True).items():
            path = checked(kernel_library.source_path(name))
            if self.workload.VERIFY_PROGRAM_PINS:
                assert sha256(path) == expected, "changed public program generator"
            self.pins[str(path)] = expected
        paths = (__file__, self.workload.__file__, self.elf, self.qemu, hal.HEADER,
                 *(tools / name for name in self.workload.PIN_PATHS))
        for filename in paths:
            path = checked(filename)
            self.pins[str(path)] = sha256(path)

    def check_memory(self):
        low, high = 0x80000000, 0x80000000 + (8 << 30)
        assert all(low <= row["address"] < row["address"] + row["memory_bytes"] <= high
                   for row in self.image.segments), "ELF exceeds actual QEMU virt8GiB DRAM"
        assert low <= self.chain.placement["arena_base"] <= high - max(
            item["total_dram_bytes"] for item in self.data["kernels"]), "arena exceeds actual QEMU DRAM"
        if self.args.corrupt_region:
            allowed = self.workload.CORRUPT_REGIONS or self.chain.layout
            assert self.args.corrupt_region in allowed
            assert 0 <= self.args.corrupt_offset < self.chain.layout[self.args.corrupt_region]["bytes"]

    def start_qemu(self):
        serial = self.uart.prepare()
        command = [str(self.qemu), "-machine", "virt", "-m", "8G", "-smp", "1",
                   "-accel", "tcg,thread=single", "-bios", "none", "-nodefaults", "-nographic",
                   "-monitor", "none", "-serial", "file:" + serial, "-S", "-gdb",
                   "unix:qemu.sock,server=on,wait=off", "-kernel", str(self.elf)]
        self.receipt["qemu_command"] = command
        self.progress("static-admitted")
        with (self.out / "qemu.log").open("wb") as qlog:
            self.process = subprocess.Popen(command, cwd=self.out, stdout=qlog,
                                            stderr=subprocess.STDOUT, start_new_session=True)
        self.progress("qemu-started")
        startup = min(self.deadline, time.monotonic() + 30)
        while not (self.out / "qemu.sock").exists():
            if self.process.poll() is not None or time.monotonic() >= startup:
                raise RuntimeError("QEMU segment startup failed")
            time.sleep(0.05)
        os.chdir(self.out)
        self.remote = Remote("qemu.sock", deadline=self.deadline,
                             timeout=min(self.args.wall_seconds, 1800))
        self.enter_guest()

    def enter_guest(self):
        remote, image = self.remote, self.image
        assert remote.request("?")[:1] in (b"S", b"T")
        remote.set_reg(32, image.entry)
        for sites in (self.ready, self.stops):
            for pc, length in sites.items():
                remote.breakpoint(pc, length)
        for pc, access in self.sites.items():
            assert remote.read(pc, access["length"]) == image.bytes_at(pc, access["length"])
            remote.breakpoint(pc, access["length"])
        self.uart.register(remote, image, set(self.ready) | set(self.stops) | set(self.sites))
        while True:
            pc, _ = remote.resume()
            if pc in self.ready:
                break
            assert self.uart.handle(remote, pc), "unexpected guest stop before initialized segment entry"
        self.receipt["host_elf_executed"] = True
        assert pc in self.ready, "guest did not reach normal initialized segment entry"
        self.initial = self.workload.initialize(remote, image)
        self.entry_pc = pc

    def restore(self):
        args, chain, remote = self.args, self.chain, self.remote
        self.progress("restoring-complete-live-state")
        chain.restore(args.segment_index, remote.write)
        for name, member in chain.boundaries[args.segment_index]["regions"].items():
            region = chain.layout[name]
            assert remote.read(region["address"], region["bytes"]) == chain.read(member), "segment restore differs"
        self.progress("live-state-restored")
        if args.corrupt_region:
            address = chain.layout[args.corrupt_region]["address"] + args.corrupt_offset
            original = remote.read(address, 1)
            remote.write(address, bytes([original[0] ^ 1]))
            self.receipt["negative_control"] = dict(
                region=args.corrupt_region, byte_offset=args.corrupt_offset, xor=1)
        remote.write(self.image.symbols[self.workload.REQUEST_SYMBOL]["address"],
                     struct.pack("<Q", args.segment_index))
        self.device = Device(remote, self.out, self.deadline, self.bundle, self.model_root,
                             cycle_cap=self.contract.full.CYCLE_CAP, launch_range=self.launch_range)
        self.receipt["fresh_device"] = True
        self.receipt["memory_backend"] = self.device.memory_backend
        self.binding = self.device.binding
        self.receipt["model_binding"] = self.binding
        self.progress("device-ready")
        remote.set_reg(32, self.entry_pc + self.ready[self.entry_pc])

    def execute_mmio(self):
        remote, device = self.remote, self.device
        access_cap = 100000 + sum(len(words) // 4 + 64 for _, words, _ in device.programs.values())
        accesses = 0
        while time.monotonic() < self.deadline:
            pc, stop = remote.resume()
            if pc in self.stops:
                break
            if self.uart.handle(remote, pc):
                continue
            assert pc in self.sites, ("unexpected segment guest stop", pc, stop)
            access = self.sites[pc]
            address = remote.reg(access["base"]) + access["offset"]
            if access["store"]:
                value = remote.reg(access["reg"]) & 0xffffffff
                starting = address == hal.ATLAS_CSR_BASE + hal.ATLAS_CSR_EXEC_CONTROL and value == 1
                if starting:
                    self.progress("executing-global-kernel-" + str(self.launch_range[0] + len(device.completed)))
                device.write32(address, value)
                if starting:
                    self.progress("kernel-completed")
            else:
                value = device.read32(address)
                if access["signed"] and value & 0x80000000:
                    value |= 0xffffffff00000000
                remote.set_reg(access["reg"], value)
            remote.set_reg(32, pc + access["length"])
            accesses += 1
            assert accesses <= access_cap, "segment MMIO access cap"
        else:
            raise TimeoutError("segment absolute wall cap")
        self.receipt.update(kernels=device.completed, mmio_accesses=accesses)
        self.workload.capture_stop(remote, self.image, self.out, pc, self.receipt)
        assert pc in self.success, self.workload.STOP_ERROR

    def verify_completion(self):
        device, bundle, chain = self.device, self.bundle, self.chain
        assert len(device.completed) == device.launch_count == self.count
        assert [row["index"] for row in device.completed] == list(range(*self.launch_range))
        for summary in device.completed:
            item = self.bundle.kernel_paths(summary["index"])
            item.update(program=bundle.read(item["original"]), expected_sha256=bundle.sums[item["expected"]])
            self.contract.verify_kernel(summary, item, summary["index"])
        self.workload.verify_state_after(self.remote, self.image, chain, self.data,
                                        self.args.segment_index, device.completed, self.initial,
                                        self.receipt, self.launch_range)
        compared = chain.compare(self.args.segment_index, self.remote.read)
        self.receipt["boundaries"] = {row["name"]: row["sha256"] for row in compared}
        self.receipt["boundary_comparisons"] = compared
        self.receipt["uart"] = verify_uart((self.out / "uart.log").read_bytes(),
                                           self.workload.RESULT_NAME, self.count,
                                           progress_launches=[self.count])
        self.verify_sources()
        assert time.monotonic() < self.deadline
        self.receipt.update(status="PASS_SEGMENT_WITH_MODEL_DEVICE", source_bookends_equal=True,
                            inputs_sha256=self.pins,
                            measured_kernel_cycles=sum(row["cycles"] for row in device.completed))

    def verify_sources(self):
        self.bundle.verify_bookends()
        self.chain.verify_bookends()
        if self.device:
            self.device.verify_bookends()
        assert all(sha256(checked(path)) == value for path, value in self.pins.items())
        assert model_binding(self.model_root) == self.binding, "model source bookends differ"

    def finish(self):
        receipt, device, process = self.receipt, self.device, self.process
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        if (self.workload.CHECK_FAILURE_SOURCES and receipt.get("full_static_admission")
                and not receipt["source_bookends_equal"]):
            try:
                self.verify_sources()
                receipt.update(source_bookends_equal=True, inputs_sha256=self.pins)
            except Exception as error:
                receipt["bookend_error"] = dict(type=type(error).__name__, message=str(error))
        receipt["runner_max_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if device and hasattr(device, "core"):
            receipt["actual_core_cycles"] = device.core.cycle_count
        receipt.setdefault("measured_kernel_cycles", sum(row["cycles"] for row in receipt["kernels"]))
        self.progress("finished")
        if process:
            try:
                status = Path(f"/proc/{process.pid}/status").read_text()
                receipt["qemu_peak_rss_kib"] = int(next(
                    line.split()[1] for line in status.splitlines() if line.startswith("VmHWM:")))
            except (FileNotFoundError, StopIteration):
                pass
            self.close(lambda: receipt.update(qemu_child_reaped=reap_qemu(process)))
        for obj in (self.remote, device):
            if obj:
                self.close(obj.close)
        if self.uart:
            receipt["uart_transport"] = self.uart.summary()
            self.close(self.uart.close)
        os.chdir(self.previous_cwd)
        receipt["wall_seconds"] = time.monotonic() - self.started
        (self.out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")

    def close(self, close):
        try:
            close()
        except Exception as error:
            self.receipt.setdefault("cleanup_errors", []).append(str(error))
            self.receipt["status"] = "FAILED"

    def execute(self):
        self.progress("static-admission")
        def interrupted(signum, frame):
            raise InterruptedError("segment interrupted by signal " + str(signum))
        signal.signal(signal.SIGTERM, interrupted)
        try:
            self.check_inputs()
            self.start_qemu()
            self.restore()
            self.execute_mmio()
            self.verify_completion()
        except Exception as error:
            self.receipt["error"] = dict(type=type(error).__name__, message=str(error))
            if self.device:
                self.receipt["kernels"] = self.device.completed
        finally:
            self.finish()
        print(json.dumps({key: self.receipt[key] for key in ("status", "error") if key in self.receipt}), flush=True)
        return 0 if self.receipt["status"].startswith("PASS_") else 1
