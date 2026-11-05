"""Run an unchanged RV64 graph ELF in generic QEMU with a diagnostic device.

QEMU is all-stop while npu_model ticks; npu_model is idle while QEMU runs.
Whole-model qualification requires a final live receipt. No SoC/FPGA binding.
"""

import argparse
from contextlib import suppress
import importlib
import json
import os
from pathlib import Path
import signal
import struct
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).absolute().parent.parent))

import kernel_library  # noqa: E402,F401
from atlas_tooling import write_json, sha as digest
from atlas_host_layout import HEADER as LAYOUT_HEADER, HEADER_SHA256
from elf_image import Image, mmio_sites
from gdb_remote import Remote
from model_device import Device, checked
from bundle import Bundle, sha, verify_completion
from result_lines import verify as verify_result_lines

FULL_CONTRACTS = {"tiny-full": "tiny_full_contract", "smol-full": "smol_full_contract"}


def resnet_stop_sites(image):
    names = ["atlas_workload_main", "atlas_block_trap_report"]
    if "finish" in image.symbols:
        names.append("finish")
    return {pc: length for name in names for pc, length, word in image.instructions(name)
            if length == 4 and word == 0x10500073}


def mmio_access_cap(device, workload):
    if workload not in FULL_CONTRACTS:
        return 100000
    # Count actual admitted IMEM writes plus bounded CSR traffic per launch.
    return 100000 + sum(len(words) // 4 + 64 for _, words, _ in device.programs.values())


def select_uart_bridge(profile, requested):
    if profile is None:
        if requested:raise ValueError('SiFive UART bridge requires an explicit full-host UART profile')
        return False
    if (profile.get('kind'),profile.get('base')) == ('ns16550',0x10000000):
        if requested:raise ValueError('SiFive UART bridge cannot replace the NS16550 profile')
        return False
    if requested and profile.get('kind') == 'sifive':
        from uart_device import validate_profile
        validate_profile(profile)
        return True
    raise ValueError('generic QEMU requires NS16550 at0x10000000 or explicit --sifive-uart-bridge')


def interrupted(signum, frame):
    raise InterruptedError("host ISA replay terminated by signal " + str(signum))


def resnet_completion(remote, image, device, out):
    mailbox = remote.read(image.symbols["atlas_workload_mailbox"]["address"], 928)
    (out / "mailbox.bin").write_bytes(mailbox)
    fields = struct.unpack_from("<10I", mailbox)
    tohost = int.from_bytes(remote.read(image.symbols["tohost"]["address"], 8), "little")
    logits = remote.read(image.symbols["logits"]["address"], 4000)
    (out / "logits.bin").write_bytes(logits)
    expected = image.bytes_at(image.symbols["graph_expected"]["address"], 4000)
    verify_completion(mailbox, tohost, logits, expected, device.completed)
    return dict(mailbox_words=fields, tohost=tohost, logits_words=1000,
        logits_sha256=digest(logits), all_logits_exact=True)


def open_bundle(args):
    workload = getattr(args, "workload", "resnet")
    contract = None
    if workload == "resnet":
        bundle = Bundle(Path(args.bundle_root).absolute())
    elif workload in FULL_CONTRACTS:
        from full_bundle import FullBundle
        bundle = FullBundle(Path(args.bundle_root).absolute(), workload)
        contract = importlib.import_module(FULL_CONTRACTS[workload])
    else:
        from block_bundle import BlockBundle
        bundle = BlockBundle(Path(args.bundle_root).absolute(), workload)
        contract = importlib.import_module(workload + "_contract")
    return workload, contract, bundle


def pin_sources(pins, contract, workload, elf, qemu):
    helpers = ["run.py", "bundle.py", "model_device.py", "elf_image.py", "gdb_remote.py", "placement.py",
               "result_lines.py"]
    if contract is not None:
        helpers.append("block_bundle.py")
        if workload in FULL_CONTRACTS:
            helpers.append("full_bundle.py")
        if workload == "tiny-full":
            helpers.append("tiny_contract.py")
        path = checked(Path(contract.__file__).absolute())
        pins[str(path)] = sha(path)
    for name in helpers:
        path = checked(Path(__file__).absolute().parent / name)
        pins[str(path)] = sha(path)
    if workload == "smol-full":
        from kernel_library import code_sources
        tools = Path(__file__).absolute().parents[1]
        for name, expected_hash in code_sources(traffic_reuse=True).items():
            path = checked(kernel_library.source_path(name))
            assert sha(path) == expected_hash, "changed full Smol generator helper"
            pins[str(path)] = expected_hash
        path = checked(tools / "smol_full_plan.py")
        pins[str(path)] = sha(path)
    arena_helper = checked(Path(__file__).absolute().parents[1] / "chipyard/arena.py")
    pins[str(arena_helper)] = sha(arena_helper)
    for path in (elf, qemu):
        pins[str(path)] = sha(path)
    pins[str(LAYOUT_HEADER)] = HEADER_SHA256
    for module in ("atlas_tooling", "atlas_path_guard", "atlas_host_layout", "run_model_tile", "atlas_tile_arithmetic"):
        path = checked(Path(sys.modules[module].__file__).absolute())
        pins[str(path)] = sha(path)


def close_resources(receipt, process, device, remote, original_cwd,
                    uart_stream, broker, out, deadline):
    def terminate_qemu():
        if not process or process.poll() is not None:
            return
        with suppress(ProcessLookupError):os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            with suppress(ProcessLookupError):os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)

    cleanup = [terminate_qemu]
    if device:
        cleanup.append(device.close)
    if remote:
        cleanup.append(remote.close)
    if original_cwd is not None:
        cleanup.append(lambda: os.chdir(original_cwd))
    if uart_stream:
        cleanup.append(uart_stream.close)
    for close in cleanup:
        try:
            close()
        except Exception as error:
            detail = dict(type=type(error).__name__, message=str(error))
            receipt.setdefault("cleanup_errors", []).append(detail)
            receipt.setdefault("error", detail)
            receipt["status"] = "FAILED"
    if broker is not None:
        receipt['uart_transport'] = broker.summary()
        receipt['uart_stream_sha256'] = sha(out/'uart.log')
    receipt["qemu_reaped"] = process is not None and process.poll() is not None
    receipt["wall_remaining_seconds"] = deadline - time.monotonic()
    write_json(out / "receipt.json", receipt)


def wait_for_qemu(process, sock, deadline):
    startup_deadline = min(deadline, time.monotonic() + 30)
    while not sock.exists():
        if process.poll() is not None:
            raise RuntimeError("QEMU exited before debugger socket")
        if time.monotonic() >= startup_deadline:
            raise TimeoutError("QEMU startup deadline")
        time.sleep(0.05)


def access_mmio(remote, device, pc, access):
    address = remote.reg(access["base"]) + access["offset"]
    if access["store"]:
        device.write32(address, remote.reg(access["reg"]) & 0xFFFFFFFF)
    else:
        value = device.read32(address)
        if access["signed"] and value & 0x80000000:
            value |= 0xFFFFFFFF00000000
        remote.set_reg(access["reg"], value)
    remote.set_reg(32, pc + access["length"])


def execute_guest(remote, device, broker, sites, stop_sites, uart_sites,
                  deadline, receipt, access_cap):
    accesses = uart_accesses = 0
    while time.monotonic() < deadline:
        receipt["host_elf_execution_attempted"] = True
        pc, stop = remote.resume()
        receipt["host_elf_executed"] = True
        if pc in stop_sites:
            break
        if pc in uart_sites:
            broker.access(remote,uart_sites[pc])
            remote.set_reg(32,pc+uart_sites[pc]['length'])
            uart_accesses += 1
            continue
        assert pc in sites, ("unexpected debugger stop", hex(pc), stop)
        access_mmio(remote, device, pc, sites[pc])
        accesses += 1
        assert accesses < access_cap, "MMIO access cap"
    else:
        raise TimeoutError("host graph wall cap")
    return accesses, uart_accesses


def completion_sites(args, workload, contract, bundle, image):
    if contract is None:
        stop_sites = resnet_stop_sites(image)
    else:
        if workload == "smol-full":
            from run_model_tile import configure_model
            _, _, assemble, _ = configure_model(checked(Path(args.model_root).absolute()))
            contract.validate_bundle(bundle, image, assemble=assemble)
        else:
            contract.validate_bundle(bundle, image)
        stop_sites = contract.stop_sites(image)
    assert stop_sites, "completion WFI instruction missing"
    return stop_sites


def install_breakpoints(remote, image, sites, uart_sites, stop_sites):
    remote.set_reg(32, image.entry)
    for pc, site in sites.items():
        assert remote.read(pc, site["length"]) == image.bytes_at(pc, site["length"])
        remote.breakpoint(pc, site["length"])
    for pc, site in uart_sites.items():
        assert remote.read(pc,site['length']) == image.bytes_at(pc,site['length'])
        remote.breakpoint(pc,site['length'])
    for pc, length in stop_sites.items():
        remote.breakpoint(pc, length)


def check_uart_sites(image, uart_profile, bridge, sites, stop_sites, pins):
    uart_sites = {}
    if bridge:
        from uart_device import sites as find_uart_sites
        helper = checked(Path(__file__).absolute().with_name('uart_device.py'))
        pins[str(helper)] = sha(helper)
        uart_sites = find_uart_sites(image,uart_profile)
        if set(uart_sites) & (set(sites) | set(stop_sites)):
            raise ValueError('UART sites overlap accelerator accessors or completion stops')
    return uart_sites


def run(args):
    out = checked(Path(args.out).absolute())
    out.mkdir(exist_ok=False)
    receipt = dict(status="FAILED", command=[sys.executable, *sys.argv],
        scope="generic RV64 ISA execution plus serialized npu_model MMIO device; no SoC/FPGA/platform qualification",
        portable_adaptation=True, full_soc_executed=False, host_elf_executed=False, verbose_rsp_trace_enabled=False,
        kernels=[])
    signal.signal(signal.SIGTERM, interrupted)
    process = remote = device = uart_stream = broker = original_cwd = None
    deadline = time.monotonic() + args.wall_seconds
    pins = {}
    try:
        workload, contract, bundle = open_bundle(args)
        receipt["workload"] = workload
        elf = checked(Path(args.elf).absolute())
        qemu = checked(Path(args.qemu).absolute())
        assert sha(elf) == bundle.sums["host/host.elf"], "ELF differs from supplied bundle"
        image = Image(elf.read_bytes())
        build = getattr(bundle, "build", {})
        uart_profile = build.get("uart")
        bridge = select_uart_bridge(uart_profile,getattr(args,'sifive_uart_bridge',False))
        sites = mmio_sites(image)
        stop_sites = completion_sites(args, workload, contract, bundle, image)
        uart_sites = check_uart_sites(image, uart_profile, bridge, sites, stop_sites, pins)
        pin_sources(pins, contract, workload, elf, qemu)
        model_root = checked(Path(args.model_root).absolute())
        sock = checked(out / "qemu.sock")
        serial_file = 'qemu-native-uart.log' if bridge else 'uart.log'
        if bridge:
            from uart_device import SifiveUART
            uart_stream = (out/'uart.log').open('wb')
            broker = SifiveUART(uart_profile,uart_stream)
        command = [str(qemu), "-machine", "virt", "-m", "8G", "-smp", "1",
            "-accel", "tcg,thread=single", "-bios", "none",
            "-nodefaults", "-nographic", "-monitor", "none", "-serial", "file:"+serial_file, "-S", "-gdb",
            "unix:qemu.sock,server=on,wait=off", "-kernel", str(elf)]
        receipt.update(qemu_command=command, inputs_sha256=pins, bundle_sha256sums_sha256=bundle.checksums_sha256,
            mmio_sites=sites, completion_breakpoints=sorted(stop_sites))
        with (out / "qemu.log").open("wb") as qlog:
            process = subprocess.Popen(command, cwd=out, stdout=qlog, stderr=subprocess.STDOUT, start_new_session=True)
            wait_for_qemu(process, sock, deadline)
            original_cwd = Path.cwd()
            os.chdir(out)
            remote = Remote("qemu.sock", trace=None, timeout=min(args.wall_seconds, 1800), deadline=deadline)
            assert remote.request("?")[:1] in (b"S", b"T")
            receipt["qemu_pid"] = process.pid
            install_breakpoints(remote, image, sites, uart_sites, stop_sites)
            cycle_cap = getattr(args, "kernel_cycle_cap", None) or (
                contract.CYCLE_CAP if contract is not None else 20000000)
            device = Device(remote, out, deadline, bundle, model_root, cycle_cap=cycle_cap)
            access_cap = mmio_access_cap(device, workload)
            receipt["kernel_cycle_cap"] = cycle_cap
            receipt["mmio_access_cap"] = access_cap
            receipt["memory_backend"] = device.memory_backend
            accesses, uart_accesses = execute_guest(
                remote, device, broker, sites, stop_sites, uart_sites, deadline, receipt, access_cap)
            completion = (resnet_completion(remote, image, device, out) if contract is None else
                contract.verify_completion(remote, image, device, bundle, out))
            if uart_profile is not None:
                uart_raw = (out / "uart.log").read_bytes()
                completion["uart"] = dict(profile=uart_profile, sha256=digest(uart_raw),
                    **verify_result_lines(uart_raw, build["result_name"], device.kernel_count,
                        progress_launches=build["result_progress_launches"]))
            if broker is not None:
                completion['uart']['bridge'] = broker.summary()
                completion['uart_mmio_accesses'] = uart_accesses
                completion['uart_sites'] = uart_sites
            bundle.verify_bookends()
            device.verify_bookends()
            assert all(sha(Path(path)) == expected_hash for path, expected_hash in pins.items())
            assert time.monotonic() < deadline, "absolute completion deadline"
            receipt.update(status="PASS_RV64_HOST_WITH_MODEL_DEVICE",
                kernels=device.completed, mmio_accesses=accesses,
                one_device_reset=True, model_binding=device.binding, device_payloads_sha256=device.pins,
                source_bookends_equal=True, **completion)
    except Exception as error:
        receipt["error"] = dict(type=type(error).__name__, message=str(error))
        if device:
            receipt["kernels"] = device.completed
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)

        close_resources(receipt, process, device, remote, original_cwd,
                        uart_stream, broker, out, deadline)
    print(json.dumps({k: receipt[k] for k in ("status", "error") if k in receipt}), flush=True)
    return 0 if receipt["status"].startswith("PASS_") else 1


def arguments(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--wall-seconds", type=int)
    parser.add_argument("--workload", choices=("resnet", "tiny", "smol", *FULL_CONTRACTS), default="resnet")
    parser.add_argument("--kernel-cycle-cap", type=int)
    parser.add_argument('--sifive-uart-bridge',action='store_true',
                        help='model admitted board SiFive UART accesses; no hardware or baud-timing claim')
    for name in ("elf", "bundle-root", "model-root", "qemu"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args(argv)
    if args.wall_seconds is None:
        if args.workload in FULL_CONTRACTS:
            parser.error("full models require an explicit forecasted --wall-seconds budget")
        args.wall_seconds = 21600
    limit = 30 * 86400 if args.workload in FULL_CONTRACTS else 86400
    if not 0 < args.wall_seconds <= limit:
        parser.error(f"--wall-seconds must be between 1 and {limit}")
    if args.kernel_cycle_cap is not None and args.kernel_cycle_cap <= 0:
        parser.error("--kernel-cycle-cap must be positive")
    return args


if __name__ == "__main__":
    if not __debug__:
        raise SystemExit("Python -O is unsupported: strict runtime checks require assertions")
    args = arguments()
    raise SystemExit(run(args))
