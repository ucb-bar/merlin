"""Small all-stop RSP client, with finite socket waits and exact transfer checks.

Protocol: https://sourceware.org/gdb/current/onlinedocs/gdb.html/Overview.html
Packets: https://sourceware.org/gdb/current/onlinedocs/gdb.html/Packets.html
"""

import kernel_library  # noqa: E402,F401
from atlas_tooling import sha
import json
import re
import socket
import time


def frame(payload):
    return b"$" + payload + b"#" + f"{sum(payload)&255:02x}".encode()


MAX_PACKET = 16 * 1024 * 1024
_SPECIAL = re.compile(rb"[}*]")
_FRAME_SPECIAL = re.compile(rb"[#}*]")


def decode_response(raw):
    if len(raw) > MAX_PACKET:
        raise ValueError("remote packet exceeds bound")
    if b"}" not in raw and b"*" not in raw:
        return bytes(raw)
    output = bytearray()
    index = 0
    while index < len(raw):
        match = _SPECIAL.search(raw, index)
        end = match.start() if match else len(raw)
        if len(output) + end-index > MAX_PACKET:
            raise ValueError("remote packet exceeds bound")
        output.extend(raw[index:end])
        if not match:
            break
        index = end+1
        if index == len(raw):
            raise ValueError("truncated escape" if raw[end] == 125 else "truncated RLE")
        value = raw[index]
        index += 1
        if raw[end] == 125:
            if len(output) == MAX_PACKET:
                raise ValueError("remote packet exceeds bound")
            output.append(value ^ 0x20)
        else:
            count = value-29
            if not output or not 3 <= count <= 97:
                raise ValueError("invalid RLE count")
            if len(output)+count > MAX_PACKET:
                raise ValueError("remote packet exceeds bound")
            output.extend(output[-1:]*count)
    return bytes(output)


class Remote:
    def __init__(self, path, trace=None, timeout=1800, deadline=None):
        self.socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.timeout = timeout
        self.deadline = deadline
        self._buffer = b""
        self._offset = 0
        self.limit_wait()
        self.socket.connect(path)
        self.trace = trace
        self.packet_limit = 1024

    def close(self):
        self.socket.close()

    def limit_wait(self):
        deadline = getattr(self, "deadline", None)
        if deadline is not None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("absolute RSP deadline")
            self.socket.settimeout(min(getattr(self, "timeout", 1800), remaining))

    def send(self, data):
        self.limit_wait()
        self.socket.sendall(data)

    def refill(self):
        self.limit_wait()
        self._buffer = self.socket.recv(65536)
        self._offset = 0
        if not self._buffer:
            raise EOFError("remote disconnected")

    def byte(self):
        if getattr(self, "_offset", 0) >= len(getattr(self, "_buffer", b"")):
            self.refill()
        result = self._buffer[self._offset:self._offset+1]
        self._offset += 1
        return result

    def receive(self):
        self.limit_wait()
        while True:
            if getattr(self, "_offset", 0) >= len(getattr(self, "_buffer", b"")):
                self.refill()
            begin = self._buffer.find(b"$", self._offset)
            if begin >= 0:
                self._offset = begin+1
                break
            self._offset = len(self._buffer)
        raw = bytearray()
        def append(chunk):
            if len(raw)+len(chunk) > MAX_PACKET:
                raise ValueError("packet exceeds bound")
            raw.extend(chunk)
        while True:
            if self._offset == len(self._buffer):
                self.refill()
            begin = scan = self._offset
            while True:
                match = _FRAME_SPECIAL.search(self._buffer, scan)
                if not match:
                    append(self._buffer[begin:])
                    self._offset = len(self._buffer)
                    break
                at = match.start()
                if self._buffer[at] == 35:
                    append(self._buffer[begin:at])
                    self._offset = at+1
                    self.limit_wait()
                    checksum = int(self.byte()+self.byte(),16)
                    if checksum != sum(raw) & 255:
                        self.send(b"-")
                        raise ValueError("remote checksum differs")
                    self.send(b"+")
                    result = decode_response(raw)
                    self.limit_wait()
                    return result
                if at+1 < len(self._buffer):
                    scan = at+2
                    continue
                append(self._buffer[begin:])
                self._offset = len(self._buffer)
                append(self.byte())
                break

    def request(self, command):
        self.limit_wait()
        command = command.encode() if isinstance(command, str) else command
        if any(c in command for c in (b"$", b"#", b"}")):
            raise ValueError("unescaped outgoing command")
        started = time.monotonic()
        for attempt in range(3):
            self.send(frame(command))
            ack = self.byte()
            if ack == b"+":
                break
            if ack != b"-":
                raise ValueError("missing remote acknowledgment")
        else:
            raise ValueError("remote refused packet repeatedly")
        reply = self.receive()
        if self.trace:
            self.trace.write(json.dumps(dict(command_prefix=command[:80].decode(errors="replace"),
                command_bytes=len(command), command_sha256=sha(command), response_bytes=len(reply),
                response_sha256=sha(reply), wall_seconds=time.monotonic() - started)) + "\n")
            self.trace.flush()
        if re.fullmatch(rb"E[0-9a-fA-F]{2}(:.*)?", reply):
            raise ValueError("remote error " + reply.decode())
        return reply

    def reg(self, index):
        assert 0 <= index <= 32
        value = self.request(f"p{index:x}")
        if not value:
            all_regs = self.request("g")
            assert len(all_regs) >= 33 * 16
            value = all_regs[index * 16 : (index + 1) * 16]
        raw = bytes.fromhex(value.decode())
        assert len(raw) == 8
        return int.from_bytes(raw, "little")

    def set_reg(self, index, value):
        assert 0 < index <= 32 and 0 <= value < 1 << 64
        value = value.to_bytes(8, "little").hex().encode()
        reply = self.request(f"P{index:x}=".encode() + value)
        if not reply:
            all_regs = self.request("g")
            assert len(all_regs) >= 33 * 16
            reply = self.request(b"G" + all_regs[: index * 16] + value + all_regs[(index + 1) * 16 :])
        assert reply == b"OK"

    def read(self, address, length):
        assert 0 <= address < 1 << 64 and 0 <= length <= 64 * 1024 * 1024 and address + length <= 1 << 64
        chunks = []
        for offset in range(0, length, self.packet_limit):
            count = min(self.packet_limit, length - offset)
            raw = bytes.fromhex(self.request(f"m{address+offset:x},{count:x}").decode())
            assert len(raw) == count, "short remote memory read"
            chunks.append(raw)
        return b"".join(chunks)

    def write(self, address, data):
        assert 0 <= address < 1 << 64 and len(data) <= 64 * 1024 * 1024 and address + len(data) <= 1 << 64
        for offset in range(0, len(data), self.packet_limit):
            block = data[offset : offset + self.packet_limit]
            assert self.request(f"M{address+offset:x},{len(block):x}:".encode() + block.hex().encode()) == b"OK"

    def breakpoint(self, address, length=2, kind=0):
        assert self.request(f"Z{kind:x},{address:x},{length:x}") == b"OK"

    def resume(self):
        response = self.request("c")
        assert response[:1] in (b"S", b"T") and response[1:3] == b"05", response
        return self.reg(32), response
