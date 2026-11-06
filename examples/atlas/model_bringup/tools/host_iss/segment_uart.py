"""Optional functional UART transport during normal segment guest startup."""
from pathlib import Path

import kernel_library  # noqa: E402,F401
from atlas_path_guard import checked
from atlas_tooling import sha256
from run import select_uart_bridge
from uart_device import SifiveUART, sites


class Transport:
    def __init__(self, profile, requested, image, out):
        self.modeled = select_uart_bridge(profile, requested)
        self.sites = sites(image, profile) if self.modeled else {}
        self.profile, self.out = dict(profile), out
        self.accesses, self.stream, self.broker = 0, None, None

    @staticmethod
    def pin(pins):
        for name in ("segment_uart.py", "uart_device.py", "run.py"):
            path = checked(Path(__file__).absolute().with_name(name))
            pins[str(path)] = sha256(path)

    def prepare(self):
        if self.modeled:
            self.stream = (self.out/"uart.log").open("wb")
            self.broker = SifiveUART(self.profile, self.stream)
        return "qemu-native-uart.log" if self.modeled else "uart.log"

    def register(self, remote, image, reserved):
        if set(self.sites) & set(reserved):
            raise ValueError("UART instruction aliases a segment/device breakpoint")
        for pc, site in self.sites.items():
            if remote.read(pc, site["length"]) != image.bytes_at(pc, site["length"]):
                raise ValueError("actual segment UART instruction differs")
            remote.breakpoint(pc, site["length"])

    def handle(self, remote, pc):
        if pc not in self.sites:
            return False
        self.broker.access(remote, self.sites[pc])
        remote.set_reg(32, pc+self.sites[pc]["length"])
        self.accesses += 1
        return True

    def summary(self):
        if self.broker:
            return dict(self.broker.summary(), mmio_accesses=self.accesses)
        if self.modeled:
            return dict(backend="modeled-sifive-uart-not-started", profile=self.profile,
                        hardware_execution=False, baud_timing=False, mmio_accesses=0)
        return dict(backend="qemu-native-ns16550", hardware_execution=False, baud_timing=False)

    def close(self):
        if self.stream:
            self.stream.close()
